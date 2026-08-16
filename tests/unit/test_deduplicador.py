"""Tests del deduplicador de mensajes (PB-004)."""

from __future__ import annotations

from src.infrastructure.external.whatsapp.deduplicador import DeduplicadorDeMensajes


def test_la_primera_vez_es_nuevo() -> None:
    dedup = DeduplicadorDeMensajes()

    assert dedup.marcar_si_es_nuevo("wamid.A") is True


def test_la_segunda_vez_es_duplicado() -> None:
    """Es el caso real: Meta reintrega el mismo mensaje."""
    dedup = DeduplicadorDeMensajes()
    dedup.marcar_si_es_nuevo("wamid.A")

    assert dedup.marcar_si_es_nuevo("wamid.A") is False


def test_mensajes_distintos_no_se_confunden() -> None:
    dedup = DeduplicadorDeMensajes()

    assert dedup.marcar_si_es_nuevo("wamid.A") is True
    assert dedup.marcar_si_es_nuevo("wamid.B") is True


def test_respeta_el_tope_de_capacidad() -> None:
    """El caché no puede crecer sin límite: sería un vector de memoria."""
    dedup = DeduplicadorDeMensajes(capacidad=3)

    for indice in range(10):
        dedup.marcar_si_es_nuevo(f"wamid.{indice}")

    assert len(dedup) == 3


def test_desaloja_el_mas_viejo_primero() -> None:
    dedup = DeduplicadorDeMensajes(capacidad=2)
    dedup.marcar_si_es_nuevo("wamid.VIEJO")
    dedup.marcar_si_es_nuevo("wamid.MEDIO")
    dedup.marcar_si_es_nuevo("wamid.NUEVO")

    # El más viejo salió, así que se vuelve a ver como nuevo.
    assert dedup.marcar_si_es_nuevo("wamid.VIEJO") is True
    assert dedup.marcar_si_es_nuevo("wamid.NUEVO") is False


def test_las_entradas_vencidas_se_purgan() -> None:
    dedup = DeduplicadorDeMensajes(ttl_segundos=0)
    dedup.marcar_si_es_nuevo("wamid.A")

    assert dedup.marcar_si_es_nuevo("wamid.A") is True
    assert len(dedup) == 1


def test_olvidar_permite_reprocesar() -> None:
    """Un fallo transitorio no debe dejar el mensaje sin respuesta para siempre."""
    dedup = DeduplicadorDeMensajes()
    dedup.marcar_si_es_nuevo("wamid.A")

    dedup.olvidar("wamid.A")

    assert dedup.marcar_si_es_nuevo("wamid.A") is True


def test_olvidar_algo_que_no_esta_no_falla() -> None:
    DeduplicadorDeMensajes().olvidar("wamid.INEXISTENTE")


def test_volver_a_ver_un_mensaje_lo_rejuvenece() -> None:
    """LRU: un duplicado reciente no debería ser el próximo en salir."""
    dedup = DeduplicadorDeMensajes(capacidad=2)
    dedup.marcar_si_es_nuevo("wamid.A")
    dedup.marcar_si_es_nuevo("wamid.B")
    dedup.marcar_si_es_nuevo("wamid.A")  # duplicado: A pasa a ser el más reciente
    dedup.marcar_si_es_nuevo("wamid.C")  # desaloja al más viejo, que ahora es B

    assert dedup.marcar_si_es_nuevo("wamid.A") is False
    assert dedup.marcar_si_es_nuevo("wamid.B") is True
